// Job metadata builder for the Raptor visual assembly (geometry comes from assembly.json).
import { createHash } from 'node:crypto';
import { readFileSync, statSync, writeFileSync } from 'node:fs';
import { buildSemiProfessionalEvidencePack } from '../../src/engine/evidence-readiness';
import type { ReferenceView } from '../../src/engine/reference-set';
import { createCanvas, loadImage } from '@napi-rs/canvas';

// Same deterministic input-fit formula as src/engine/reference.ts (product assets), from real pixels.
async function analyse(file: string) {
  const image = await loadImage(`${process.cwd()}/${dir}/sources/${file}`);
  const canvas = createCanvas(48, 48); const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0, 48, 48);
  const px = ctx.getImageData(0, 0, 48, 48).data; let r = 0, g = 0, b = 0, w = 0;
  for (let i = 0; i < px.length; i += 4) { const a = px[i + 3] / 255; r += px[i] * a; g += px[i + 1] * a; b += px[i + 2] * a; w += a; }
  r /= Math.max(w, 1); g /= Math.max(w, 1); b /= Math.max(w, 1);
  const ratio = image.width / image.height;
  const ratioScore = ratio >= 0.65 && ratio <= 1.85 ? 1 : ratio >= 0.4 && ratio <= 2.4 ? 0.72 : 0.4;
  const resolutionScore = Math.min(1, Math.min(image.width, image.height) / 1200);
  const hex = [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');
  return { width: image.width, height: image.height, averageColor: `#${hex}`,
    brightness: (r * 0.2126 + g * 0.7152 + b * 0.0722) / 255, fit: Math.round((ratioScore * 0.62 + resolutionScore * 0.38) * 100) };
}

const dir = 'work/raptor';
const sha = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
async function view(id: string, file: string, sourceType: ReferenceView['sourceType'],
  role: ReferenceView['role'], covered: ReferenceView['coveredRoles'], capabilities: ReferenceView['capabilities'],
  notes: string[]): Promise<ReferenceView> {
  const { width, height, averageColor, brightness, fit } = await analyse(file);
  return { id, assetKind: 'product', url: `blob:${id}`, fileName: file, fileSize: statSync(`${dir}/sources/${file}`).size,
    mimeType: 'image/png', lastModified: 1, role, coveredRoles: covered, capabilities, sourceType,
    evidence: { fileName: file, width, height, averageColor, brightness, portraitSuitability: fit, notes } };
}
const design = await view('design-drawing', 'design.png', 'cad', 'measurement', ['left', 'right', 'front'],
  ['shape', 'depth', 'scale', 'interfaces'], ['Orthographic side/front of the Xacro collision boxes at zero pose, 1 px = 1 mm.',
    'Design specification, not a physical measurement.']);
const concept = await view('concept-image', 'concept.png', 'photo', 'material', ['left'], ['shape', 'surface'],
  ['AI-generated concept image; styling and material reference only.']);
const pad = (id: string, property: string, valueMm: number) => ({ id, property, valueMm, toleranceMm: 0.5,
  status: 'measured' as const, sourceViewId: 'design-drawing' });
const evidencePack = buildSemiProfessionalEvidencePack([design, concept], {
  profile: 'product-visualization',
  dimensions: [pad('pad-length', 'foot_pad_length', 150), pad('pad-width', 'foot_pad_width', 100),
    pad('metatarsus-length', 'metatarsus_length', 240)],
  sourceAudits: [
    { viewId: 'design-drawing', provenance: 'client-measured', geometryConsistency: 'verified', dimensionLegibility: 'verified' },
    { viewId: 'concept-image', provenance: 'synthetic-concept', geometryConsistency: 'partial', dimensionLegibility: 'unreadable' }],
});
const assembly = JSON.parse(readFileSync(`${dir}/assembly.json`, 'utf8'));
const contract = (id: string, componentId: string, axis: 'x' | 'y' | 'z', expectedMm: number) => ({ id, label: id,
  target: { kind: 'component', componentId }, axis, measurement: 'size', space: 'component-local', expectedMm, toleranceMm: 0.5,
  evidence: { status: 'measured', sourceViewId: 'view_01', source: 'measured on the Xacro-derived CAD drawing' } });
assembly.dimensionContracts = [contract('pad-length', 'left_foot_link__pad', 'x', 150),
  contract('pad-width', 'left_foot_link__pad', 'z', 100), contract('metatarsus-length', 'left_foot_link__metatarsus', 'y', 240)];
const job = { schema: 'morphloom.job/0.1', id: 'raptor-digitigrade-visual', target: 'review',
  request: '10축 디지티그레이드 랩터 로봇 제품의 링크별 부품 어셈블리 시각 외형을 목표 콘셉트 스타일로 상세 제작 (물리 형상은 Xacro 기준)',
  sources: [{ viewId: 'view_01', path: `${dir}/sources/design.png`, sha256: sha(`${dir}/sources/design.png`) },
    { viewId: 'view_02', path: `${dir}/sources/concept.png`, sha256: sha(`${dir}/sources/concept.png`) }],
  evidencePack, assembly };
writeFileSync(`${dir}/job.json`, JSON.stringify(job, null, 1));
console.log('job written', assembly.components.length, 'components; readiness', JSON.stringify(evidencePack.readiness).slice(0, 300));
