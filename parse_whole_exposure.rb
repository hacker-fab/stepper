#Based on parse_exposure from the EDA team in the S25
#This would be run from KLayout following the EDA team's SOP for mask generation
#with the exception that parse_whole_exposure would be run instead of parse_exposure

include RBA

LAYERS = (1..8).map { |i| RBA::LayerInfo.new(i, 0) }

cv = RBA::CellView::active
layout = cv.layout
top = cv.cell
dbu = layout.dbu
output_dir = File.dirname(cv.filename || ".")

template_name = "Template"
frame_layer_info = RBA::LayerInfo.new(8, 0)  # Template frame is drawn here
frame_layer_index = layout.layer(frame_layer_info)

# Map LayerInfo to layout indices
layer_indices = {}
LAYERS.each do |li|
  idx = layout.layer(li)
  layer_indices[li] = idx unless idx == -1
end

# Step 1: Get Template instance bounding boxes from (8,0) frame
template_instances = []
template_bboxes = []

top.each_inst do |inst|
  next unless inst.is_pcell?
  next unless inst.cell.name == template_name

  frame_shape = nil
  inst.cell.shapes(frame_layer_index).each do |s|
    if s.is_box?
      frame_shape = s
      break
    end
  end

  next unless frame_shape
  bbox = frame_shape.box.transformed(inst.trans)

  template_instances << inst
  template_bboxes << bbox
end

puts "Found #{template_instances.size} Template instances with (8,0) frames"

# Step 2: Flatten layout once
flat_layout = layout.dup
flat_top = flat_layout.top_cell
flat_top.flatten(true)

# Step 3: Merge all template bounding boxes
template_region = RBA::Region::new
template_bboxes.each { |bbox| template_region.insert(bbox) }
merged_region = template_region.merged

puts "Exporting merged design from #{template_bboxes.size} Template instances..."

# Step 4: Build new layout
new_layout = RBA::Layout.new
new_layout.dbu = dbu
new_cell = new_layout.create_cell("TOP")

layer_indices.each do |li, src_layer_idx|
  dst_layer_idx = new_layout.layer(li)

  region = RBA::Region::new
  flat_top.shapes(src_layer_idx).each_overlapping(merged_region.bbox) do |shape|
    next unless shape.is_box? || shape.is_path? || shape.is_polygon?

    geom = shape.is_box? ? shape.box : shape.is_path? ? shape.path : shape.polygon

    # Keep shape only if it intersects the merged region
    if !(RBA::Region::new(geom) & merged_region).is_empty?
      region.insert(geom)
    end
  end

  # Special handling for (8,0) frames: keep each individual template frame
  if li.layer == 8 && li.datatype == 0
    region.each do |g|
      template_bboxes.each do |bbox|
        new_cell.shapes(dst_layer_idx).insert(g) if g.bbox == bbox
      end
    end
  else
    region.each { |g| new_cell.shapes(dst_layer_idx).insert(g) }
  end
end

# Step 5: Add the merged outer border (yellow = 99/0)
box_layer = new_layout.layer(RBA::LayerInfo.new(99, 0))
merged_region.each { |poly| new_cell.shapes(box_layer).insert(poly) }

# Step 6: Write result
file_path = File.join(output_dir, "full_design.gds")
new_layout.write(file_path)

puts "Done. Exported merged design to #{file_path}"